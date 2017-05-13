
"""
Fibonacci heap.

File: fibonacci_heap_mod.py
Author: Keith Schwarz (htiek@cs.stanford.edu)
Ported to Python by Dan Stromberg (strombrg@gmail.com)

An implementation of a priority queue backed by a Fibonacci heap, as described
by Fredman and Tarjan.  Fibonacci heaps are interesting theoretically because
they have asymptotically good runtime guarantees for many operations.  In
particular, insert, peek, and decrease-key all run in amortized O(1) time.
dequeue_min and delete each run in amortized O(lg n) time.  This allows
algorithms that rely heavily on decrease-key to gain significant performance
boosts.  For example, Dijkstra's algorithm for single-source shortest paths can
be shown to run in O(m + n lg n) using a Fibonacci heap, compared to O(m lg n)
using a standard binary or binomial heap.

Internally, a Fibonacci heap is represented as a circular, doubly-linked list
of trees obeying the min-heap property.  Each node stores pointers to its
parent (if any) and some arbitrary child.  Additionally, every node stores its
degree (the number of children it has) and whether it is a "marked" node.
Finally, each Fibonacci heap stores a pointer to the tree with the minimum
value.

To insert a node into a Fibonacci heap, a singleton tree is created and merged
into the rest of the trees.  The merge operation works by simply splicing
together the doubly-linked lists of the two trees, then updating the min
pointer to be the smaller of the minima of the two heaps.  Peeking at the
smallest element can therefore be accomplished by just looking at the min
element.  All of these operations complete in O(1) time.

The tricky operations are dequeue_min and decrease_key.  dequeue_min works by
removing the root of the tree containing the smallest element, then merging its
children with the topmost roots.  Then, the roots are scanned and merged so
that there is only one tree of each degree in the root list.  This works by
maintaining a dynamic array of trees, each initially null, pointing to the
roots of trees of each dimension.  The list is then scanned and this array is
populated.  Whenever a conflict is discovered, the appropriate trees are merged
together until no more conflicts exist.  The resulting trees are then put into
the root list.  A clever analysis using the potential method can be used to
show that the amortized cost of this operation is O(lg n), see "Introduction to
Algorithms, Second Edition" by Cormen, Rivest, Leiserson, and Stein for more
details.

The other hard operation is decrease_key, which works as follows.  First, we
update the key of the node to be the new value.  If this leaves the node
smaller than its parent, we're done.  Otherwise, we cut the node from its
parent, add it as a root, and then mark its parent.  If the parent was already
marked, we cut that node as well, recursively mark its parent, and continue
this process.  This can be shown to run in O(1) amortized time using yet
another clever potential function.  Finally, given this function, we can
implement delete by decreasing a key to -infinity, then calling dequeue_min to
extract it.
"""

import math
import collections


def merge_lists(one, two):
    """
    Merge 2 lists.

    Utility function which, given two pointers into disjoint circularly-
    linked lists, merges the two lists together into one circularly-linked
    list in O(1) time.  Because the lists may be empty, the return value
    is the only pointer that's guaranteed to be to an element of the
    resulting list.

    This function assumes that one and two are the minimum elements of the
    lists they are in, and returns a pointer to whichever is smaller.  If
    this condition does not hold, the return value is some arbitrary pointer
    into the doubly-linked list.

    @param one A reference to one of the two deques.
    @param two A reference to the other of the two deques.
    @return A reference to the smallest element of the resulting list.
    """
    # There are four cases depending on whether the lists are None or not.
    # We consider each separately.
    if one is None and two is None:
        # Both None, resulting list is None.
        return None
    elif one is not None and two is None:
        # Two is None, result is one.
        return one
    elif one is None and two is not None:
        # One is None, result is two.
        return two
    else:
        # Both non-None; actually do the splice.

        # This is actually not as easy as it seems.  The idea is that we'll
        # have two lists that look like this:
        #
        # +----+     +----+     +----+
        # |    |--N->|one |--N->|    |
        # |    |<-P--|    |<-P--|    |
        # +----+     +----+     +----+
        #
        #
        # +----+     +----+     +----+
        # |    |--N->|two |--N->|    |
        # |    |<-P--|    |<-P--|    |
        # +----+     +----+     +----+
        #
        # And we want to relink everything to get
        #
        # +----+     +----+     +----+---+
        # |    |--N->|one |     |    |   |
        # |    |<-P--|    |     |    |<+ |
        # +----+     +----+<-\  +----+ | |
        #                  \  P        | |
        #                   N  \       N |
        # +----+     +----+  \->+----+ | |
        # |    |--N->|two |     |    | | |
        # |    |<-P--|    |     |    | | P
        # +----+     +----+     +----+ | |
        #              ^ |             | |
        #              | +-------------+ |
        #              +-----------------+

        # Cache this since we're about to overwrite it.
        one_next = one.m_next

        one.m_next = two.m_next
        one.m_next.m_prev = one
        two.m_next = one_next
        two.m_next.m_prev = two

        # Return a pointer to whichever's smaller.
        if one.m_priority < two.m_priority:
            return one
        else:
            return two


def merge(one, two):
    """
    Merge 2 Fibonacci heaps.

    Given two Fibonacci heaps, returns a new Fibonacci heap that contains
    all of the elements of the two heaps.  Each of the input heaps is
    destructively modified by having all its elements removed.  You can
    continue to use those heaps, but be aware that they will be empty
    after this call completes.

    @param one The first Fibonacci heap to merge.
    @param two The second Fibonacci heap to merge.
    @return A new Fibonacci_heap containing all of the elements of both
            heaps.
    """
    # Create a new Fibonacci_heap to hold the result.
    result = Fibonacci_heap()

    # Merge the two Fibonacci heap root lists together.  This helper function
    # also computes the min of the two lists, so we can store the result in
    # the m_min field of the new heap.
    result.m_min = merge_lists(one.m_min, two.m_min)

    # The size of the new heap is the sum of the sizes of the input heaps.
    result.m_size = one.m_size + two.m_size

    # Clear the old heaps.
    one.m_size = two.m_size = 0
    one.m_min = None
    two.m_min = None

    # Return the newly-merged heap.
    return result


# In order for all of the Fibonacci heap operations to complete in O(1),
# clients need to have O(1) access to any element in the heap.  We make
# this work by having each insertion operation produce a handle to the
# node in the tree.  In actuality, this handle is the node itself.
class Entry(object):
    # pylint: disable=too-many-instance-attributes

    """Hold an entry in the heap."""

    __slots__ = ['m_degree', 'm_is_marked', 'm_parent', 'm_child', 'm_next', 'm_prev', 'm_elem', 'm_priority']

    def __init__(self, elem, priority):
        """Initialize an Entry in the heap."""
        # Number of children
        self.m_degree = 0
        # Whether this node is marked
        self.m_is_marked = False

        # Parent in the tree, if any.
        self.m_parent = None

        # Child node, if any.
        self.m_child = None

        self.m_next = self.m_prev = self
        self.m_elem = elem
        self.m_priority = priority

    def __lt__(self, other):
        """Return True iff self's priority is less than other's."""
        return self.m_priority < other.m_priority

    def __eq__(self, other):
        """Return True iff ==."""
        if self.m_priority == other.m_priority:
            return True
        else:
            return self.m_elem == other.m_elem

    def __gt__(self, other):
        """Return True iff >."""
        if self.m_priority > other.m_priority:
            return True
        else:
            return self.m_elem > other.m_elem

    def __cmp__(self, other):
        """Python 2.x-style comparison."""
        if self.__lt__(other):
            return -1
        elif self.__gt__(other):
            return 1
        else:
            return 0

#    def __cmp__(self, other):
#        """
#        Comparison method, 2.x style.
#        We compare object identity, rather than priority and value
#        """
#        if id(self) == id(other):
#            return 0
#        elif id(self) < id(other):
#            return -1
#        else:
#            return 1
#
#    def __lt__(self, other):
#        """Comparison method, 3.x style"""
#        if self.__cmp__(other) == -1:
#            return True
#        else:
#            return False
#
#    def __eq__(self, other):
#        """Comparison method, 3.x style"""
#        if self.__cmp__(other) == 0:
#            return True
#        else:
#            return False

    def get_value(self):
        """
        Return the element represented by this heap entry.

        @return The element represented by this heap entry.
        """
        return self.m_elem

    def set_value(self, value):
        """
        Set the element associated with this heap entry.

        @param value The element to associate with this heap entry.
        """
        self.m_elem = value

    def get_priority(self):
        """
        Return the priority of this element.

        @return The priority of this element.
        """
        return self.m_priority

    def _entry(self, elem, priority):
        """
        Construct a new Entry that holds the given element with the indicated priority.

        @param elem The element stored in this node.
        @param priority The priority of this element.
        """
        self.m_next = self.m_prev = self
        self.m_elem = elem
        self.m_priority = priority


class Fibonacci_heap(object):

    """
    A class representing a Fibonacci heap.

    @author Keith Schwarz (htiek@cs.stanford.edu)
    """

    def __init__(self):
        """Initialize the fibonacci heap."""
        # Pointer to the minimum element in the heap.
        self.m_min = None

        # Cached size of the heap, so we don't have to recompute this explicitly.
        self.m_size = 0

    def enqueue(self, value, priority):
        """
        Insert the specified element into the Fibonacci heap with the specified priority.

        Its priority must be a valid double, so you cannot set the priority to NaN.

        @param value The value to insert.
        @param priority Its priority, which must be valid.
        @return An Entry representing that element in the tree.
        """
        self._check_priority(priority)

        # Create the entry object, which is a circularly-linked list of length
        # one.
        result = Entry(value, priority)

        # Merge this singleton list with the tree list.
        self.m_min = merge_lists(self.m_min, result)

        # Increase the size of the heap; we just added something.
        self.m_size += 1

        # Return the reference to the new element.
        return result

    def min(self):
        """
        Return an Entry object corresponding to the minimum element of the Fibonacci heap.

        Raise an IndexError if the heap is empty.

        @return The smallest element of the heap.
        @raises IndexError If the heap is empty.
        """
        if not bool(self):
            raise IndexError("Heap is empty.")
        return self.m_min

    def __bool__(self):
        """
        Return whether the heap is nonempty.

        @return Whether the heap is nonempty.
        """
        return self.m_min is not None

    __nonzero__ = __bool__

    def __len__(self):
        """
        Return the number of elements in the heap.

        @return The number of elements in the heap.
        """
        return self.m_size

    def dequeue_min(self):
        # pylint: disable=too-many-branches
        """
        Dequeue and return the minimum element of the Fibonacci heap.

        If the heap is empty, this throws an IndexError.

        @return The smallest element of the Fibonacci heap.
        @raises IndexError if the heap is empty.
        """
        # Check for whether we're empty.
        if not bool(self):
            raise IndexError("Heap is empty.")

        # Otherwise, we're about to lose an element, so decrement the number of
        # entries in this heap.
        self.m_size -= 1

        # Grab the minimum element so we know what to return.
        min_elem = self.m_min

        # Now, we need to get rid of this element from the list of roots.  There
        # are two cases to consider.  First, if this is the only element in the
        # list of roots, we set the list of roots to be None by clearing m_min.
        # Otherwise, if it's not None, then we write the elements next to the
        # min element around the min element to remove it, then arbitrarily
        # reassign the min.
        if self.m_min.m_next is self.m_min:
            # Case one
            self.m_min = None
        else:
            # Case two
            self.m_min.m_prev.m_next = self.m_min.m_next
            self.m_min.m_next.m_prev = self.m_min.m_prev
            # Arbitrary element of the root list.
            self.m_min = self.m_min.m_next

        # Next, clear the parent fields of all of the min element's children,
        # since they're about to become roots.  Because the elements are
        # stored in a circular list, the traversal is a bit complex.
        if min_elem.m_child is not None:
            # Keep track of the first visited node.
            curr = min_elem.m_child
            while True:
                curr.m_parent = None

                # Walk to the next node, then stop if this is the node we
                # started at.
                curr = curr.m_next
                if curr is min_elem.m_child:
                    # This was a do-while (curr != minElem.mChild);
                    break

        # Next, splice the children of the root node into the topmost list,
        # then set self.m_min to point somewhere in that list.
        self.m_min = merge_lists(self.m_min, min_elem.m_child)

        # If there are no entries left, we're done.
        if self.m_min is None:
            return min_elem

        # Next, we need to coalesce all of the roots so that there is only one
        # tree of each degree.  To track trees of each size, we allocate an
        # ArrayList where the entry at position i is either None or the
        # unique tree of degree i.
        tree_table = collections.deque()

        # We need to traverse the entire list, but since we're going to be
        # messing around with it we have to be careful not to break our
        # traversal order mid-stream.  One major challenge is how to detect
        # whether we're visiting the same node twice.  To do this, we'll
        # spent a bit of overhead adding all of the nodes to a list, and
        # then will visit each element of this list in order.
        to_visit = collections.deque()

        # To add everything, we'll iterate across the elements until we
        # find the first element twice.  We check this by looping while the
        # list is empty or while the current element isn't the first element
        # of that list.

        # for (Entry<T> curr = self.m_min; toVisit.isEmpty() || toVisit.get(0) != curr; curr = curr.m_next)
        curr = self.m_min
        while not to_visit or to_visit[0] is not curr:
            to_visit.append(curr)
            curr = curr.m_next

        # Traverse this list and perform the appropriate unioning steps.
        for curr in to_visit:
            # Keep merging until a match arises.
            while True:
                # Ensure that the list is long enough to hold an element of this
                # degree.
                while curr.m_degree >= len(tree_table):
                    tree_table.append(None)

                # If nothing's here, we can record that this tree has this size
                # and are done processing.
                if tree_table[curr.m_degree] is None:
                    tree_table[curr.m_degree] = curr
                    break

                # Otherwise, merge with what's there.
                other = tree_table[curr.m_degree]
                # Clear the slot
                tree_table[curr.m_degree] = None

                # Determine which of the two trees has the smaller root, storing
                # the two trees accordingly.
                # minimum = (other.m_priority < curr.m_priority)? other : curr
                if other.m_priority < curr.m_priority:
                    minimum = other
                else:
                    minimum = curr
                # maximum = (other.m_priority < curr.m_priority)? curr  : other
                if other.m_priority < curr.m_priority:
                    maximum = curr
                else:
                    maximum = other

                # Break max out of the root list, then merge it into min's child
                # list.
                maximum.m_next.m_prev = maximum.m_prev
                maximum.m_prev.m_next = maximum.m_next

                # Make it a singleton so that we can merge it.
                maximum.m_next = maximum.m_prev = maximum
                minimum.m_child = merge_lists(minimum.m_child, maximum)

                # Reparent maximum appropriately.
                maximum.m_parent = minimum

                # Clear maximum's mark, since it can now lose another child.
                maximum.m_is_marked = False

                # Increase minimum's degree; it now has another child.
                minimum.m_degree += 1

                # Continue merging this tree.
                curr = minimum

            # Update the global min based on this node.  Note that we compare
            # for <= instead of < here.  That's because if we just did a
            # reparent operation that merged two different trees of equal
            # priority, we need to make sure that the min pointer points to
            # the root-level one.
            if curr.m_priority <= self.m_min.m_priority:
                self.m_min = curr
        return min_elem

    def decrease_key(self, entry, new_priority):
        """
        Decrease the key of the specified element to the new priority.

        If the new priority is greater than the old priority, this function raises an ValueError.  The new priority must
        be a finite double, so you cannot set the priority to be NaN, or +/- infinity.  Doing so also raises an
        ValueError.

        It is assumed that the entry belongs in this heap.  For efficiency reasons, this is not checked at runtime.

        @param entry The element whose priority should be decreased.
        @param new_priority The new priority to associate with this entry.
        @raises ValueError If the new priority exceeds the old
                priority, or if the argument is not a finite double.
        """
        self._check_priority(new_priority)
        if new_priority > entry.m_priority:
            raise ValueError("New priority exceeds old.")

        # Forward this to a helper function.
        self.decrease_key_unchecked(entry, new_priority)

    def delete(self, entry):
        """
        Delete this Entry from the Fibonacci heap that contains it.

        It is assumed that the entry belongs in this heap.  For efficiency
        reasons, this is not checked at runtime.

        @param entry The entry to delete.
        """
        # Use decreaseKey to drop the entry's key to -infinity.  This will
        # guarantee that the node is cut and set to the global minimum.
        self.decrease_key_unchecked(entry, float("-inf"))

        # Call dequeue_min to remove it.
        self.dequeue_min()

    @staticmethod
    def _check_priority(priority):
        """
        Utility function: given a user-specified priority, check whether it's a valid double and throw a ValueError otherwise.

        @param priority The user's specified priority.
        @raises ValueError if it is not valid.
        """
        if math.isnan(priority) or math.isinf(priority):
            raise ValueError("Priority {} is invalid.".format(priority))

    def decrease_key_unchecked(self, entry, priority):
        """
        Decrease the key of a node in the tree without doing any checking to ensure that the new priority is valid.

        @param entry The node whose key should be decreased.
        @param priority The node's new priority.
        """
        # First, change the node's priority.
        entry.m_priority = priority

        # If the node no longer has a higher priority than its parent, cut it.
        # Note that this also means that if we try to run a delete operation
        # that decreases the key to -infinity, it's guaranteed to cut the node
        # from its parent.
        if entry.m_parent is not None and entry.m_priority <= entry.m_parent.m_priority:
            self.cut_node(entry)

        # If our new value is the new min, mark it as such.  Note that if we
        # ended up decreasing the key in a way that ties the current minimum
        # priority, this will change the min accordingly.
        if entry.m_priority <= self.m_min.m_priority:
            self.m_min = entry

    def cut_node(self, entry):
        """
        Cut a node from its parent.

        If the parent was already marked, recursively cuts that node from its parent as well.

        @param entry The node to cut from its parent.
        """
        # Begin by clearing the node's mark, since we just cut it.
        entry.m_is_marked = False

        # Base case: If the node has no parent, we're done.
        if entry.m_parent is None:
            return

        # Rewire the node's siblings around it, if it has any siblings.
        if entry.m_next is not entry:
            # Has siblings
            entry.m_next.m_prev = entry.m_prev
            entry.m_prev.m_next = entry.m_next

        # If the node is the one identified by its parent as its child,
        # we need to rewrite that pointer to point to some arbitrary other
        # child.
        if entry.m_parent.m_child is entry:
            if entry.m_next is not entry:
                # If there are any other children, pick one of them arbitrarily.
                entry.m_parent.m_child = entry.m_next
            else:
                # Otherwise, there aren't any children left and we should clear the
                # pointer and drop the node's degree.
                entry.m_parent.m_child = None

        # Decrease the degree of the parent, since it just lost a child.
        entry.m_parent.m_degree -= 1

        # Splice this tree into the root list by converting it to a singleton
        # and invoking the merge subroutine.
        entry.m_prev = entry.m_next = entry
        self.m_min = merge_lists(self.m_min, entry)

        # Mark the parent and recursively cut it if it's already been
        # marked.
        if entry.m_parent.m_is_marked:
            self.cut_node(entry.m_parent)
        else:
            entry.m_parent.m_is_marked = True

        # Clear the relocated node's parent; it's now a root.
        entry.m_parent = None
